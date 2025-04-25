class R(dict):
    """返回数据的统一格式类"""

    def __init__(self):
        super().__init__()
        self['code'] = 0

    @staticmethod
    def error(code=500, msg="未知异常，请联系管理员"):
        """错误返回"""
        return R().update({"code": code, "msg": msg})

    @staticmethod
    def ok(msg="成功", data=None):
        """成功返回"""
        response = R()
        response.update({"msg": msg})
        if data:
            response.update({"data": data})
        return response

    def update(self, values):
        """更新字典内容"""
        super().update(values)
        return self
